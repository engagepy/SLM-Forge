"""Finding, importing, preparing, writing and reviewing training data."""

from pathlib import Path

from sqlmodel import Session, select

from slm.data import format as fmt
from slm.data import scout_tools
from slm.data.pipeline import build_feedback_version, preview_mapping
from slm.db import (
    DatasetVersion,
    PreferencePair,
    Project,
    SftExample,
    awaiting_review,
    count,
    engine,
)
from slm.db import (
    count as count_rows,
)
from slm.events import canvas_changed
from slm.models import manage
from slm.tuner.tools._core import (
    MAX_SYNTHETIC_PER_CALL,
    MAX_SYNTHETIC_TOTAL,
    MAX_UNREVIEWED,
    Ctx,
    _own_dataset,
    _spend,
    _submit,
    _version_brief,
    _wait,
    tool,
)
from slm.tuner.util import clip as _clip


@tool
def search_datasets(ctx: Ctx, query: str) -> list[dict]:
    """Search Hugging Face datasets. Returns id, licence, size, tasks and a description."""
    return _clip(scout_tools.search_datasets(query, limit=8), 240)


@tool
def preview_dataset(ctx: Ctx, repo_id: str) -> dict:
    """Look at a Hub dataset's columns and first rows before importing it, plus a suggested
    column mapping. Always preview before importing."""
    data = scout_tools.preview_rows(repo_id, n=3)
    data["suggested_mapping"] = fmt.guess_mapping(data["columns"])
    return _clip(data, 400)


@tool
def scout_datasets(ctx: Ctx, brief: str, reason: str = "") -> dict:
    """Delegate the dataset hunt to DataScout, a specialist agent: it runs several searches,
    previews candidates in parallel, reads their cards and returns a ranked shortlist (licence,
    columns, suggested mapping, answer length, fit score 0–10, caveats) with a best pick or null.
    brief: the goal, the target output format, the target answer length, whether the user might
    ship the model (licence), and anything to avoid. Costs a handful of API calls; outside a round
    it's a card. Then import_dataset the pick and plan_preparation it."""
    from slm.tuner import specialists

    pid = ctx.context.project_id
    if proposal := _spend(
        pid, "scout_datasets", "scout", "Have DataScout find public datasets", reason,
        {"brief": brief[:200]}, {"brief": brief},
    ):  # fmt: skip
        return proposal
    report, calls = specialists.scout(pid, brief)
    return {"searched_and_previewed": len(calls), **report.model_dump()}


@tool
def plan_preparation(ctx: Ctx, dataset_id: int, brief: str, reason: str = "") -> dict:
    """Delegate the mapping and cleaning plan for an imported dataset to DataPrep, a specialist
    agent: it inspects rows, tries a mapping against them and returns mapping, min/max answer
    chars, max_seq_length and the expected kept fraction. brief: the project's output format and
    target answer length. Then call prepare_dataset with the plan (and max_examples per your plan)."""
    from slm.tuner import specialists

    pid = ctx.context.project_id
    with Session(engine()) as s:
        ds = _own_dataset(s, pid, dataset_id)
    if proposal := _spend(
        pid, "plan_preparation", "prep", f"Have DataPrep plan the cleaning of {ds.name}", reason,
        {"dataset": ds.name, "rows": ds.n_rows}, {"dataset_id": dataset_id, "brief": brief},
    ):  # fmt: skip
        return proposal
    plan, calls = specialists.prep(pid, f"dataset_id: {dataset_id}\n{brief}")
    return {"checked": len(calls), **plan.model_dump()}


@tool
def import_dataset(
    ctx: Ctx, repo_id: str, max_rows: int = 20000, config: str | None = None, split: str = "train", reason: str = ""
) -> dict:
    """Import rows from a Hugging Face dataset into the project (streamed; up to 200,000 rows; the
    canvas shows progress). Import generously: prepare_dataset(max_examples=...) samples the pool
    down to the plan's target, and later rounds can draw a fresh sample. Waits for the import to
    finish. Outside a round the user set in motion this is a proposal they confirm first; give a
    plain-words `reason` for the card."""
    max_rows = max(1, min(int(max_rows), 200_000))
    args = {"repo_id": repo_id, "max_rows": max_rows, "config": config, "split": split}
    if proposal := _spend(
        ctx.context.project_id, "import_dataset", "import", f"Import {repo_id}", reason,
        {"repo_id": repo_id, "max_rows": max_rows}, args,
    ):  # fmt: skip
        return proposal
    # The licence travels with the data into the export's model card, so record it at import.
    job = _submit(
        "import_dataset",
        {"repo_id": repo_id, "max_rows": max_rows, "config": config, "split": split,
         "license": scout_tools.dataset_license(repo_id)},
        ctx.context.project_id,
    )  # fmt: skip
    job = _wait(job.id, 3600)
    if job.status != "succeeded":
        return {"status": job.status, "error": job.error or "still running", "job_id": job.id}
    return {"status": "imported", **_clip(job.result)}


@tool
def inspect_dataset(ctx: Ctx, dataset_id: int) -> dict:
    """Columns, row count, three sample rows and a suggested mapping for an imported or uploaded dataset."""
    with Session(engine()) as s:
        ds = _own_dataset(s, ctx.context.project_id, dataset_id)
    rows = scout_tools.read_raw(Path(ds.raw_path), limit=3)
    return _clip(
        {
            "name": ds.name,
            "rows": ds.n_rows,
            "columns": ds.columns,
            "sample": rows,
            "suggested_mapping": fmt.guess_mapping(ds.columns),
        },
        400,
    )


@tool
def prepare_dataset(
    ctx: Ctx,
    dataset_id: int,
    format: str,
    prompt_column: str | None = None,
    response_column: str | None = None,
    extra_input_column: str | None = None,
    messages_column: str | None = None,
    text_column: str | None = None,
    chosen_column: str | None = None,
    rejected_column: str | None = None,
    constant_system_prompt: str | None = None,
    max_seq_length: int = 1024,
    max_examples: int | None = None,
    min_answer_chars: int = 2,
    max_chars: int | None = None,
    long_examples: str = "auto",
    seed: int = 0,
) -> dict:
    """Turn raw rows into clean training records: map columns, fix encoding, remove duplicates and
    junk, split train/valid/test and measure token lengths. Waits for the result.

    format: instruction (prompt_column + response_column, optional extra_input_column),
            chat (messages_column), text (text_column), preference (prompt/chosen/rejected columns).
    A column value may be a constant written as "=text", and can include {column} placeholders,
    e.g. prompt_column="=How do I make {title}?".
    Examples longer than max_seq_length are handled here rather than silently truncated during
    training (which cuts off the end of answers): long_examples="auto" drops over-long Q&A/chat
    examples and splits long raw text into windows; "keep" leaves them to be truncated.
    max_examples: after cleaning and deduplication, train on a random sample of this many (the
    plan's data target); the rest of the import stays on disk as a pool for later rounds, and a
    different seed draws a fresh sample from it. min_answer_chars / max_chars are DataPrep's plan:
    records with a shorter answer, or longer than max_chars in total, are dropped.
    The result's "length" report shows the length distribution and how many didn't fit: if a
    large share was dropped, raise max_seq_length (check memory with plan_training) or pick data
    with shorter examples."""
    mapping = {"format": format}
    for key, value in (
        ("prompt", prompt_column),
        ("response", response_column),
        ("input", extra_input_column),
        ("messages", messages_column),
        ("text", text_column),
        ("chosen", chosen_column),
        ("rejected", rejected_column),
    ):
        if value:
            mapping[key] = value
    if constant_system_prompt:
        mapping["system"] = "=" + constant_system_prompt
    with Session(engine()) as s:
        ds = _own_dataset(s, ctx.context.project_id, dataset_id)
    check = preview_mapping(ds, mapping, n=1)
    if check["failed"] == check["sampled"]:
        return {"status": "mapping doesn't work", "errors": check["errors"], "columns": ds.columns}
    job = _submit(
        "prepare_dataset",
        {
            "dataset_id": dataset_id,
            "mapping": mapping,
            "max_seq_length": max_seq_length,
            "max_examples": max_examples,
            "rules": {"min_chars": min_answer_chars, "long_examples": long_examples}
            | ({"max_chars": max_chars} if max_chars else {}),
            "seed": seed,
        },
        ctx.context.project_id,
    )
    job = _wait(job.id, 600)
    if job.status != "succeeded":
        return {"status": job.status, "error": job.error or "still running", "job_id": job.id}
    with Session(engine()) as s:
        v = s.get(DatasetVersion, job.result["dataset_version_id"])
        return {
            "status": "prepared",
            **_version_brief(v),
            "sample_record": _clip(preview_mapping(ds, mapping, n=1)["records"][:1], 500),
        }


@tool
def generate_synthetic_examples(ctx: Ctx, kind: str, count: int, focus: str, reason: str = "") -> dict:
    """Have the teacher model (GPT-6) write a SMALL set of training examples: kind "sft" (question +
    ideal answer) or "preference" (ideal vs. weak answer). At most 50 per call and 200 per project:
    the dataset itself comes from public data (scout_datasets → import_dataset → prepare_dataset).
    Two uses: a seed of a few dozen when nothing public fits, and a targeted top-up for a gap the
    evaluation showed. Examples wait unapproved; spot-check them with review_synthetic_examples,
    then approve. This costs API calls (one per example). Outside a round the user set in motion
    it's a proposal they confirm first; give a plain-words `reason` for the card."""
    pid = ctx.context.project_id
    count = max(1, min(count, MAX_SYNTHETIC_PER_CALL))
    with Session(engine()) as s:
        unreviewed = sum(
            count_rows(s, m, m.project_id == pid, *awaiting_review(m)) for m in (PreferencePair, SftExample)
        )
        written = sum(
            count_rows(s, m, m.project_id == pid, m.source == "synthetic") for m in (PreferencePair, SftExample)
        )
    if written + count > MAX_SYNTHETIC_TOTAL:
        raise ValueError(
            f"This project already has {written} synthetic examples; the limit is {MAX_SYNTHETIC_TOTAL}. The teacher "
            "model writes seeds and top-ups only. Get the dataset from public data instead: scout_datasets, "
            "import_dataset, then prepare_dataset(max_examples=...)."
        )
    if unreviewed >= MAX_UNREVIEWED:
        raise ValueError(
            f"{unreviewed} synthetic examples are still waiting for review. Review and approve (or reject) "
            "them with review_synthetic_examples before writing more."
        )
    args = {"kind": kind, "count": count, "focus": focus}
    if proposal := _spend(
        pid, "generate_synthetic_examples", "synthesize",
        f"Write {count} {'preference pairs' if kind == 'preference' else 'training examples'}",
        reason, {"kind": kind, "count": count, "focus": focus[:200]}, args,
    ):  # fmt: skip
        return proposal
    job = _submit("synthesize", args, pid)
    job = _wait(job.id, 1800)
    return {"status": job.status, **(job.result or {}), "error": job.error or None}


def _example_ref(ref: str) -> tuple[type, int]:
    """Pairs and SFT examples live in separate tables with overlapping ids: "p12" vs "s12"."""
    kind, num = ref[:1].lower(), ref[1:]
    if kind not in ("p", "s") or not num.isdigit():
        raise ValueError(f"bad example id {ref!r}; use ids like 'p12' or 's7' as listed")
    return (PreferencePair if kind == "p" else SftExample), int(num)


@tool
def review_synthetic_examples(
    ctx: Ctx,
    approve_ids: list[str] | None = None,
    reject_ids: list[str] | None = None,
    approve_all: bool = False,
    show: int = 6,
) -> dict:
    """List synthetic examples awaiting review and approve or reject them. Ids look like "p12"
    (preference pair) or "s7" (SFT example). Reject any that are wrong or off-goal, then
    approve_all=True approves everything else still pending. Read a sample before approving."""
    pid = ctx.context.project_id
    rejected = {_example_ref(r) for r in reject_ids or []}
    changed = 0
    with Session(engine()) as s:
        for model, i in rejected:
            if (row := s.get(model, i)) and row.project_id == pid and row.used_in_job_id is None:
                s.delete(row)
                changed += 1
        for model, i in (_example_ref(r) for r in approve_ids or []):
            if (row := s.get(model, i)) and row.project_id == pid:
                row.approved = True
                s.add(row)
                changed += 1
        if approve_all:
            for model in (PreferencePair, SftExample):
                for row in s.exec(select(model).where(model.project_id == pid, *awaiting_review(model))).all():
                    if (model, row.id) not in rejected:
                        row.approved = True
                        s.add(row)
                        changed += 1
        s.commit()
        pairs, sft = (
            s.exec(select(m).where(m.project_id == pid, *awaiting_review(m)).limit(show)).all()
            for m in (PreferencePair, SftExample)
        )
        pending = sum(count(s, m, m.project_id == pid, *awaiting_review(m)) for m in (PreferencePair, SftExample))
    canvas_changed(pid)
    return {
        "changed": changed,
        "still_pending": pending,
        "sample_pairs": [
            {
                "id": f"p{p.id}",
                "prompt": _clip(p.prompt, 150),
                "chosen": _clip(p.chosen, 250),
                "rejected": _clip(p.rejected, 150),
            }
            for p in pairs
        ],
        "sample_sft": [{"id": f"s{e.id}", "messages": _clip(e.messages, 250)} for e in sft],
    }


@tool
def build_dataset_from_examples(ctx: Ctx, max_seq_length: int = 1024) -> dict:
    """Turn every approved, not-yet-used SFT example (the user's rewritten answers plus approved
    synthetic examples) into a prepared dataset version you can train on. Examples longer than
    max_seq_length are dropped here rather than truncated by the trainer; read the "length" report
    and, if many were dropped, write shorter examples or raise max_seq_length (check memory)."""
    pid = ctx.context.project_id
    with Session(engine()) as s:
        project = s.get(Project, pid)
    v = build_feedback_version(pid, model_path=manage.serving_path(project), max_seq_length=max_seq_length)
    canvas_changed(pid)
    return {"status": "prepared", **_version_brief(v)}
