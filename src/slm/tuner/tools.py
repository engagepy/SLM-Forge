"""The Tuner's hands: every action it can take on the platform.

Each tool wraps existing, tested backend code and returns a compact dict the agent can reason
about. Quick jobs (dataset import, preparation, synthesis) are awaited inside the tool. Runs (downloads,
training, export) are only proposed: the user confirms them (tuner/confirm.py), and the Tuner is
woken when they're confirmed and again when they finish.
"""

import asyncio
import functools
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from agents import RunContextWrapper, function_tool
from sqlalchemy.orm.attributes import flag_modified
from sqlmodel import Session, select

from slm import hardware, profile
from slm.agents.provider import get_provider
from slm.data import format as fmt
from slm.data import scout_tools
from slm.data.pipeline import build_feedback_version, preview_mapping
from slm.db import (
    Checkpoint,
    Dataset,
    DatasetVersion,
    Feedback,
    Job,
    Metric,
    PreferencePair,
    Project,
    SftExample,
    awaiting_review,
    count,
    engine,
    now,
    ready,
    studio_state,
)
from slm.db import (
    count as count_rows,
)
from slm.db import test_cases as db_test_cases
from slm.events import canvas_changed, shutting_down
from slm.feedback import record_feedback
from slm.inference.engine import EngineBusy, SamplingParams
from slm.inference.engine import engine as infer
from slm.models import hub, manage
from slm.sessions import overview, resume_project, stop_project
from slm.train.config import TrainConfig, preset
from slm.train.worker import worker
from slm.tuner import confirm

STAGES = ("goal", "data", "model", "train", "evaluate", "refine", "export")
# Jobs the Tuner starts and does not wait for: it gets woken when they finish.
NOTIFY_KINDS = {"download", "sft", "dpo", "export"}


@dataclass
class TunerContext:
    project_id: int


Ctx = RunContextWrapper[TunerContext]


def tool(fn):
    """Register a Tuner tool. Tools block (HTTP, waiting on jobs, generating on the GPU), so each
    runs in a worker thread to keep the shared event loop, and the agent's streaming, responsive.
    functools.wraps keeps the signature and docstring the SDK builds the tool schema from."""

    @functools.wraps(fn)
    async def run_in_thread(*args, **kwargs):
        return await asyncio.to_thread(fn, *args, **kwargs)

    return function_tool(run_in_thread, strict_mode=False)


# ── helpers ─────────────────────────────────────────────────────────────────


def _clip(v, n: int = 300):
    if isinstance(v, str):
        return v if len(v) <= n else v[:n] + "…"
    if isinstance(v, list):
        return [_clip(x, n) for x in v[:12]] + (["…"] if len(v) > 12 else [])
    if isinstance(v, dict):
        return {k: _clip(x, n) for k, x in v.items()}
    return v


def _submit(kind: str, config: dict, project_id: int) -> Job:
    from slm.tuner.session import tuner

    if tuner.is_halted(project_id):
        raise ValueError(confirm.STOPPED)
    job = worker.submit(kind, config | {"origin": "tuner", "notify": kind in NOTIFY_KINDS}, project_id)
    canvas_changed(project_id)
    return job


# Tools that spend (API tokens, downloads, GPU time). Inside a round the user set in motion they run
# freely, because reaching the goal is the mandate. Outside one (the project is finished, or autopilot
# is paused) each call is a proposal the user confirms; confirming grants that one call.
MAX_UNREVIEWED = 150  # synthetic examples nobody has looked at yet: review before writing more


def _spend(pid: int, tool_name: str, kind: str, title: str, reason: str, details: dict, args: dict) -> dict | None:
    """None: go ahead. A dict: the proposal to return instead (the user decides)."""
    from slm.tuner.session import tuner

    if tuner.is_halted(pid):
        raise ValueError(confirm.STOPPED)
    if confirm.take_grant(pid, tool_name, args):
        return None
    if confirm.pending(pid):
        raise ValueError(
            "A proposal is already waiting for the user's decision. Don't start other costly work meanwhile: "
            "answer them, and wait for their go-ahead."
        )
    if not confirm.needs_go_ahead(pid):
        return None
    if not reason.strip():
        raise ValueError(
            f"{tool_name} costs something and the user hasn't set a round in motion, so it needs their "
            "go-ahead: call it again with a plain-words `reason` for the card (what it's for, what it costs)."
        )
    return confirm.propose(pid, kind, title, reason, details, {"tool": tool_name, "args": args})


def _check_stop(pid: int) -> None:
    """Between costly calls in a loop: stop when the user halted or the server is going down."""
    from slm.tuner.session import tuner

    if shutting_down.is_set() or tuner.is_halted(pid):
        raise ValueError(confirm.STOPPED)


def _wait(job_id: int, timeout: float) -> Job:
    """Wait for a quick job; the UI shows its progress meanwhile."""
    deadline = time.time() + timeout
    while True:
        with Session(engine()) as s:
            job = s.get(Job, job_id)
            if job.status in ("succeeded", "failed", "cancelled") or time.time() > deadline:
                return job
        if shutting_down.wait(1):
            return job  # the server is stopping: don't keep a thread alive for this


def _job_brief(j: Job) -> dict:
    out = {"job_id": j.id, "kind": j.kind, "status": j.status}
    if j.error:
        out["error"] = j.error[:300]
    r = j.result or {}
    for k in (
        "progress",
        "metrics",
        "warnings",
        "dataset_version_id",
        "dataset_id",
        "rows",
        "path",
        "size_gb",
        "min_ram_gb",
    ):
        if k in r:
            out[k] = r[k]
    return out


# ── situational awareness ───────────────────────────────────────────────────


def _machine_guide(hw) -> dict:
    """What this Mac can train comfortably, so plans are sized to it rather than to a textbook."""
    budget = hw.budget_gb
    tiers = [(0.6, "0.5B"), (1.6, "1.5B"), (3.5, "3B"), (8.0, "7B")]
    largest = hardware.max_params_for_budget(budget, 4) / 1e9
    comfortable = next((label for cap, label in reversed(tiers) if cap <= largest * 0.6), "0.5B")
    return {
        "ml_budget_gb": round(budget, 1),
        "largest_trainable_4bit": f"{largest:.1f}B",
        "comfortable_tier": comfortable,
        "note": "Above the comfortable tier, use preset safe, batch 1–2 and sequence length ≤ 512; "
        "the memory estimate underestimates 3B+ by ~30%.",
    }


def _estimated_minutes(pid: int, iterations: int, params_b: float, seq_len: int, batch: int) -> dict:
    """Minutes a run should take: from this Mac's last run of similar size when there is one,
    else a rule of thumb for 4-bit LoRA on Apple Silicon (calibrated at 512 tokens × batch 4)."""
    with Session(engine()) as s:
        last = s.exec(
            select(Metric)
            .join(Job, Job.id == Metric.job_id)
            .where(Job.project_id == pid, Job.kind.in_(["sft", "dpo"]), Metric.split == "train")
            .order_by(Metric.id.desc())
            .limit(20)
        ).all()
    rates = [m.values.get("it_per_sec") for m in last if m.values.get("it_per_sec")]
    if rates:
        rate, basis = sorted(rates)[len(rates) // 2], "this project's last run"
    else:
        rate = 1.3 if params_b <= 0.7 else 0.8 if params_b <= 1.6 else 0.45 if params_b <= 3.5 else 0.2
        rate *= (512 * 4) / max(1, seq_len * batch)  # rough: time scales with tokens per step
        basis = "rule of thumb for this size"
    return {"minutes": max(1, round(iterations / rate / 60)), "basis": basis}


@tool
def get_status(ctx: Ctx) -> dict:
    """Everything about the project right now: hardware, goal, base model, datasets, prepared
    versions, running and recent jobs, trained checkpoints (with metrics and warnings), feedback
    counts and exports. Call this first in every new turn if you're unsure where things stand."""
    pid = ctx.context.project_id
    hw = hardware.detect()
    with Session(engine()) as s:
        p = s.get(Project, pid)
        st = studio_state(s, pid)
        datasets = s.exec(select(Dataset).where(Dataset.project_id == pid)).all()
        versions = s.exec(select(DatasetVersion).where(DatasetVersion.project_id == pid)).all()
        jobs = s.exec(select(Job).where(Job.project_id == pid).order_by(Job.id.desc()).limit(8)).all()
        ckpts = s.exec(select(Checkpoint).where(Checkpoint.project_id == pid).order_by(Checkpoint.id)).all()
        served_id = _served_checkpoint_id(s, p)
        job_results = {j.id: j.result for j in s.exec(select(Job).where(Job.id.in_([c.job_id for c in ckpts]))).all()}

        feedback = {
            "judgements": count(s, Feedback, Feedback.project_id == pid),
            "preference_pairs_ready": count(
                s, PreferencePair, PreferencePair.project_id == pid, *ready(PreferencePair)
            ),
            "sft_examples_ready": count(s, SftExample, SftExample.project_id == pid, *ready(SftExample)),
            "synthetic_awaiting_review": sum(
                count(s, m, m.project_id == pid, *awaiting_review(m)) for m in (PreferencePair, SftExample)
            ),
            "comparisons_waiting_for_user": sum(1 for c in st.comparisons if c.get("status") == "pending"),
        }
        exports = s.exec(
            select(Job).where(Job.project_id == pid, Job.kind == "export", Job.status == "succeeded")
        ).all()
        return {
            "hardware": {
                "chip": hw.chip,
                "memory_gb": hw.total_memory_gb,
                "ml_budget_gb": round(hw.budget_gb, 1),
                "largest_trainable_4bit_params_b": round(hardware.max_params_for_budget(hw.budget_gb, 4) / 1e9, 1),
            },
            "project": {
                "name": p.name,
                "goal": p.goal,
                "system_prompt": p.system_prompt,
                "plan": p.plan,
                "test_cases": db_test_cases(p),
            },
            "this_mac": _machine_guide(hw),
            "evaluations": [
                {k: e.get(k) for k in ("id", "target", "checkpoint_id", "mean", "at")} for e in st.evals[-6:]
            ],
            "models_already_on_this_mac": [
                {k: m[k] for k in ("repo_id", "params_b", "bits", "fit")} for m in manage.local_models()
            ],
            "canvas_stage": st.stage,
            "waiting_for_user_to_confirm": (st.pending_action or {}).get("title"),
            "base_model": {
                "repo_id": p.base_model,
                "downloaded": bool(p.base_model and manage.local_path_for(p.base_model)),
                "serving": "trained adapter"
                if p.current_adapter_path
                else ("fused model" if p.current_model_path and ckpts else "base model"),
            },
            "datasets": [
                {
                    "dataset_id": d.id,
                    "name": d.name,
                    "source": d.source_ref or d.source,
                    "rows": d.n_rows,
                    "columns": d.columns,
                    "license": d.license,
                }
                for d in datasets
            ],
            "prepared_versions": [
                {
                    "version_id": v.id,
                    "kind": v.kind,
                    "train": v.n_train,
                    "valid": v.n_valid,
                    "test": v.n_test,
                    "p95_tokens": (v.token_stats or {}).get("p95"),
                    "over_max_len": (v.token_stats or {}).get("over_max_seq_length"),
                    "cleaning": (v.cleaning_report or {}).get("dropped"),
                }
                for v in versions
            ],
            "recent_jobs": [_job_brief(j) for j in jobs],
            "checkpoints": [
                {
                    "checkpoint_id": c.id,
                    "kind": c.kind,
                    "job_id": c.job_id,
                    "from_base": c.parent_id is None and i > 0,
                    "metrics": c.metrics,
                    "warnings": [w["code"] for w in (job_results.get(c.job_id) or {}).get("warnings", [])],
                    "serving": c.id == served_id,
                }
                for i, c in enumerate(ckpts)
            ],
            "feedback": feedback,
            "exports": [
                {
                    "path": j.result.get("path"),
                    "size_gb": j.result.get("size_gb"),
                    "min_ram_gb": j.result.get("min_ram_gb"),
                    "system_prompt_built_in": bool(j.result.get("system_prompt_built_in")),
                }
                for j in exports
            ],
        }


@tool
def set_stage(ctx: Ctx, stage: str, note: str) -> str:
    """Move the live canvas to a pipeline stage and show one short line about what's happening.
    stage: goal | data | model | train | evaluate | refine | export. Call it whenever the work moves on."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    with Session(engine()) as s:
        st = studio_state(s, ctx.context.project_id)
        st.stage, st.note, st.updated_at = stage, note[:200], now()
        s.add(st)
        s.commit()
    canvas_changed(ctx.context.project_id)
    return f"canvas → {stage}"


@tool
def update_project(
    ctx: Ctx,
    name: str | None = None,
    goal: str | None = None,
    system_prompt: str | None = None,
    plan: dict | None = None,
    test_cases: list[dict | str] | None = None,
) -> dict:
    """Record the project's name, goal (one or two sentences), the system prompt the model will be
    trained and used with, your plan, and the test set. Set the first three as soon as you
    understand the goal; write the plan before any data.
    plan: {task_type: persona|qa|extraction|classification|other, output_format, model_tier,
    data_target (how many examples and why), eval_design (how many cases, scored how),
    stop_rule (what score or condition ends the work)}. It's shown to the user, who can steer it.
    test_cases: [{input, expected?, kind?}]. input is what a user would type. expected is the exact
    correct output when one exists (JSON, a label, a number): those are scored by exact match
    first, so give them for every deterministic task. kind: on-goal | should-not (the right answer
    is the empty/negative result) | edge. Fixed once training starts: evaluate_model scores every
    checkpoint on the same set. Replaces the whole set."""
    pid = ctx.context.project_id
    with Session(engine()) as s:
        p = s.get(Project, pid)
        if name:
            p.name = name[:60]
        if goal:
            p.goal = goal
        if system_prompt is not None:
            p.system_prompt = system_prompt
        if plan is not None:
            p.plan = {k: str(v)[:400] for k, v in plan.items() if v is not None}
        if test_cases is not None:
            cases = []
            for c in test_cases[:60]:
                c = {"input": c} if isinstance(c, str) else dict(c)
                if str(c.get("input", "")).strip():
                    cases.append(
                        {
                            "input": str(c["input"]).strip()[:600],
                            "expected": (str(c["expected"]).strip()[:2000] if c.get("expected") else None),
                            "kind": c.get("kind") if c.get("kind") in ("on-goal", "should-not", "edge") else "on-goal",
                        }
                    )
            p.test_questions = cases
        s.add(p)
        s.commit()
        s.refresh(p)
    canvas_changed(pid)
    return {
        "name": p.name,
        "goal": p.goal,
        "system_prompt": p.system_prompt,
        "plan": p.plan,
        "test_cases": db_test_cases(p),
    }


# ── base model ──────────────────────────────────────────────────────────────


@tool
def find_base_models(ctx: Ctx, query: str = "", max_params_billion: float = 3.0, task_type: str | None = None) -> dict:
    """Base models for this project. "recommended": a curated catalog of families built as small
    models (with licence, what each is best for, why, and whether it's already on this Mac),
    ordered by suitability to task_type (persona | qa | extraction | classification | other) then
    size; every entry exists as an MLX 4-bit build. "more_from_hub": a Hub search for `query`, for
    when the user names a model. Shortlist 2–3 from recommended, then choose_base_model."""
    from slm.models import catalog

    local = {m["repo_id"]: m for m in manage.local_models()}
    budget = hardware.detect().budget_gb
    recommended = []
    for c in catalog.recommend(task_type, max_params_b=max_params_billion):
        train_gb = hub.rough_training_gb(int(c["params_b"] * 1e9), 4)
        recommended.append(
            c
            | {
                "train_memory_gb": round(train_gb, 2),
                "fit": hub.fit_verdict(train_gb, budget),
                "already_on_this_mac": c["repo_id"] in local,
            }
        )
    recommended.sort(key=lambda m: (not m["suited_to_task"], not m["already_on_this_mac"], m["params_b"]))
    rows = hub.search_models(query, max_params_b=max_params_billion, limit=12) if query else []
    found = {
        m.id: {
            "repo_id": m.id,
            "params_b": round(m.params / 1e9, 2),
            "bits": m.bits,
            "train_memory_gb": m.train_estimate_gb,
            "fit": m.fit,
            "downloads": m.downloads,
            "already_on_this_mac": m.id in local,
        }
        for m in rows
    }
    words = [w for w in query.lower().split() if len(w) > 2]
    for rid, m in local.items():  # local models matching the query, even if the Hub search missed them
        if (
            rid not in found
            and m["params_b"] <= max_params_billion
            and (not words or any(w in rid.lower() for w in words))
        ):
            found[rid] = {k: m[k] for k in ("repo_id", "params_b", "bits", "train_memory_gb", "fit")} | {
                "already_on_this_mac": True
            }
    more = sorted(found.values(), key=lambda m: (not m["already_on_this_mac"], m["params_b"]))
    return {
        "recommended": recommended,
        "more_from_hub": [m for m in more if m["repo_id"] not in {r["repo_id"] for r in recommended}],
    }


@tool
def choose_base_model(ctx: Ctx, repo_id: str, reason: str) -> dict:
    """Propose a base model. The user confirms it on a card; only then is it set (and downloaded
    if it isn't on this Mac yet). reason: one or two plain sentences on why this model, shown on
    the card. Only possible before any training has happened."""
    pid = ctx.context.project_id
    with Session(engine()) as s:
        if confirm.base_model_locked(s, s.get(Project, pid), repo_id):
            raise ValueError(confirm.SWITCH_BASE)
    on_mac = manage.local_path_for(repo_id) is not None or manage.register_local(repo_id) is not None
    info = next((m for m in manage.local_models() if m["repo_id"] == repo_id), None) if on_mac else None
    details = {"repo_id": repo_id, "on_this_mac": on_mac}
    if info:
        details |= {"params_b": info["params_b"], "bits": info["bits"]}
    title = f"Use {repo_id.split('/')[-1]}" + ("" if on_mac else " (download it)")
    return confirm.propose(pid, "model", title, reason, details, {"repo_id": repo_id})


# ── data ────────────────────────────────────────────────────────────────────


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
        ds = s.get(Dataset, dataset_id)
        if ds is None or ds.project_id != pid:
            raise ValueError("no such dataset in this project")
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
    job = _submit(
        "import_dataset",
        {"repo_id": repo_id, "max_rows": max_rows, "config": config, "split": split},
        ctx.context.project_id,
    )
    job = _wait(job.id, 3600)
    if job.status != "succeeded":
        return {"status": job.status, "error": job.error or "still running", "job_id": job.id}
    return {"status": "imported", **_clip(job.result)}


@tool
def inspect_dataset(ctx: Ctx, dataset_id: int) -> dict:
    """Columns, row count, three sample rows and a suggested mapping for an imported or uploaded dataset."""
    with Session(engine()) as s:
        ds = s.get(Dataset, dataset_id)
        if ds is None or ds.project_id != ctx.context.project_id:
            raise ValueError("no such dataset in this project")
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
        ds = s.get(Dataset, dataset_id)
        if ds is None or ds.project_id != ctx.context.project_id:
            raise ValueError("no such dataset in this project")
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
            "version_id": v.id,
            "train": v.n_train,
            "valid": v.n_valid,
            "test": v.n_test,
            "cleaning": v.cleaning_report.get("dropped"),
            "tokens": {k: v.token_stats.get(k) for k in ("p50", "p95", "max", "over_max_seq_length", "estimated")},
            "length": v.cleaning_report.get("length"),
            "sampled": v.cleaning_report.get("sampled"),
            "sample_record": _clip(preview_mapping(ds, mapping, n=1)["records"][:1], 500),
        }


# ── training ────────────────────────────────────────────────────────────────


def _plan(
    pid: int, mode: str, preset_name: str, dataset_version_id: int | None, **overrides
) -> tuple[TrainConfig, dict, DatasetVersion | None]:
    """The training config for these settings, and what the Tuner needs to know about it."""
    with Session(engine()) as s:
        project = s.get(Project, pid)
        version = s.get(DatasetVersion, dataset_version_id) if dataset_version_id else None
    path = manage.serving_path(project)
    if not path:
        raise ValueError("Download a base model first")
    shape = manage.read_shape(path)
    typical = (version.token_stats or {}).get("p95") if version else None
    cfg = preset(preset_name, shape, mode, typical)
    cfg = TrainConfig(**(cfg.model_dump() | {k: v for k, v in overrides.items() if v is not None}))
    est = cfg.memory_estimate(shape, typical)
    info = {
        "memory_gb": round(est.total_gb, 2),
        "budget_gb": round(est.budget_gb, 1),
        "fits": est.fits,
        "learning_rate": cfg.learning_rate,
        "lora_rank": cfg.lora_rank,
        "layers_tuned": cfg.num_layers,
        "batch_size": cfg.batch_size,
        "grad_accumulation": cfg.grad_accumulation_steps,
        "max_seq_length": cfg.max_seq_length,
        "grad_checkpoint": cfg.grad_checkpoint,
    }
    if version:
        iters = cfg.total_iters(version.n_train)
        info |= {
            "iterations": iters,
            "epochs": cfg.epochs_for(version.n_train),
            "train_examples": version.n_train,
            "estimated": _estimated_minutes(pid, iters, shape.params / 1e9, cfg.max_seq_length, cfg.batch_size),
        }
    return cfg, info, version


@tool
def plan_training(
    ctx: Ctx,
    mode: str = "sft",
    dataset_version_id: int | None = None,
    preset_name: str = "balanced",
    learning_rate: float | None = None,
    epochs: float | None = None,
    iters: int | None = None,
    lora_rank: int | None = None,
    max_seq_length: int | None = None,
    batch_size: int | None = None,
) -> dict:
    """Work out a training configuration without starting it: memory needed vs. this Mac's budget,
    iterations and epochs, and the key settings. mode: sft | dpo. preset_name: safe | balanced | quality.
    Guidance: for SFT, learning_rate 1e-4 is a good default (2e-4 has diverged on these models);
    1–3 epochs; max_seq_length should cover the data's p95 tokens. For DPO use lr ~5e-6, 1–2 epochs."""
    _, info, _ = _plan(
        ctx.context.project_id, mode, preset_name, dataset_version_id,
        learning_rate=learning_rate, epochs=epochs, iters=iters, lora_rank=lora_rank,
        max_seq_length=max_seq_length, batch_size=batch_size,
    )  # fmt: skip
    return info


@tool
def start_training(
    ctx: Ctx,
    mode: str = "sft",
    dataset_version_id: int | None = None,
    preset_name: str = "balanced",
    learning_rate: float | None = None,
    epochs: float | None = None,
    iters: int | None = None,
    lora_rank: int | None = None,
    max_seq_length: int | None = None,
    batch_size: int | None = None,
    start_from: str = "current",
    reason: str = "",
) -> dict:
    """Propose a training run. It starts only when the user confirms it on a card; you'll be told
    when they do, and again when it finishes (progress shows live on the canvas). mode sft needs
    dataset_version_id; mode dpo trains on the feedback pairs (needs at least 3). start_from:
    current (continue from the served model) | base (fresh run). reason: one or two plain
    sentences, shown on the card, on what this run should teach the model and roughly how long it takes."""
    pid = ctx.context.project_id
    if mode not in ("sft", "dpo"):
        raise ValueError("mode must be sft or dpo")
    if mode == "sft" and not dataset_version_id:
        raise ValueError("SFT needs dataset_version_id (prepare a dataset first)")
    cfg, info, version = _plan(
        pid, mode, preset_name, dataset_version_id,
        learning_rate=learning_rate, epochs=epochs, iters=iters, lora_rank=lora_rank,
        max_seq_length=max_seq_length, batch_size=batch_size,
    )  # fmt: skip
    if mode == "sft" and version is None:
        raise ValueError("SFT needs dataset_version_id (prepare a dataset first)")
    if not info["fits"]:
        return {"status": "won't fit", **info, "hint": "use preset safe, lower batch_size or max_seq_length"}
    config = {"train": cfg.model_dump(), "start_from": start_from}
    if version:
        config["dataset_version_id"] = version.id
    ahead = overview()["capacity"]["gpu_running"]
    if ahead:
        info["would_queue_behind"] = ahead
    title = (
        f"Train on {version.n_train} examples ({info['iterations']} steps)"
        if mode == "sft" and version
        else "Refine with a preference round (DPO)"
    )
    return confirm.propose(pid, mode, title, reason, info, {"config": config}) | info


@tool
def training_progress(ctx: Ctx, job_id: int) -> dict:
    """Current state of a job: status, iteration, latest train/validation loss, memory and warnings."""
    with Session(engine()) as s:
        job = s.get(Job, job_id)
        if job is None or job.project_id != ctx.context.project_id:
            raise ValueError("no such job in this project")
        metrics = s.exec(select(Metric).where(Metric.job_id == job_id).order_by(Metric.iteration)).all()
    train = [m for m in metrics if m.split == "train"]
    val = [m for m in metrics if m.split == "val"]
    out = _job_brief(job)
    if train:
        out["latest_train"] = {
            "iteration": train[-1].iteration,
            **{k: round(v, 4) for k, v in train[-1].values.items()},
        }
        out["first_train_loss"] = round(train[0].values["loss"], 4)
    if val:
        out["val_loss_curve"] = [round(m.values["loss"], 4) for m in val]
    return out


@tool
def cancel_job(ctx: Ctx, job_id: int) -> str:
    """Stop a queued or running job."""
    return worker.cancel(job_id)


# ── evaluation & feedback ───────────────────────────────────────────────────


def _where(project: Project, target: str) -> dict:
    """Which weights answer: current (served) | base (untrained) | checkpoint:<id>."""
    if target == "base":
        path = manage.local_path_for(project.base_model or "")
        if not path:
            raise ValueError("The base model isn't downloaded yet")
        return {"model_path": path, "adapter_path": None}
    if target.startswith("checkpoint:"):
        with Session(engine()) as s:
            c = s.get(Checkpoint, int(target.split(":", 1)[1]))
        if c is None or c.project_id != project.id:
            raise ValueError("no such checkpoint in this project")
        if c.fused_path:
            return {"model_path": c.fused_path, "adapter_path": None}
        return {"model_path": c.base_model_path, "adapter_path": c.adapter_path}
    path = manage.serving_path(project)
    if not path:
        raise ValueError("No model downloaded yet")
    return {"model_path": path, "adapter_path": project.current_adapter_path}


def _served_checkpoint_id(s: Session, project: Project) -> int | None:
    """The checkpoint the project serves right now, or None for the plain base model."""
    from slm.train.jobs import served_checkpoint

    c = served_checkpoint(s, project)
    return c.id if c else None


def _generate(
    project: Project, prompt: str, temperature: float, max_tokens: int, target: str = "current", seed: int | None = None
) -> dict:
    where = _where(project, target)
    messages = [{"role": "user", "content": prompt}]
    if project.system_prompt:
        messages.insert(0, {"role": "system", "content": project.system_prompt})
    try:
        text, stats = infer.generate(
            messages, SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=seed), **where
        )
    except EngineBusy as e:
        raise ValueError(f"The GPU is busy ({e}); try again when the job finishes.") from e
    return {"text": text.strip(), "tokens_per_sec": stats.get("tokens_per_sec"), "finish": stats.get("finish_reason")}


# Two samplings of the same prompt: a steady one and an adventurous one, so they differ enough to compare.
PARAMS_A, PARAMS_B = {"temperature": 0.7}, {"temperature": 1.05}


def _two_answers(project: Project, prompt: str, max_tokens: int) -> tuple[str, str]:
    return tuple(
        _generate(project, prompt, p["temperature"], max_tokens, seed=random.randint(0, 10**9))["text"]
        for p in (PARAMS_A, PARAMS_B)
    )


@tool
def try_model(
    ctx: Ctx, prompts: list[str], target: str = "current", temperature: float = 0.7, max_tokens: int = 300
) -> list[dict]:
    """Ask the local model a few questions and see its answers (also shown on the canvas), to show
    the user what it sounds like. For a score, use evaluate_model. target: current (served) | base
    (untrained) | checkpoint:<id>. Use 2–4 prompts; each one loads the model onto the GPU."""
    pid = ctx.context.project_id
    with Session(engine()) as s:
        project = s.get(Project, pid)
    results = []
    for q in prompts[:6]:
        r = _generate(project, q, temperature, max_tokens, target)
        results.append({"prompt": q, "target": target, **r})
    with Session(engine()) as s:
        st = studio_state(s, pid)
        st.samples = (results + list(st.samples))[:12]
        flag_modified(st, "samples")
        s.add(st)
        s.commit()
    canvas_changed(pid)
    return results


SCORE_SYSTEM = """You grade one answer from a small language model that is being fine-tuned for a
specific goal. Score it 0–10 for how well it serves that goal: correctness first (a confident wrong
fact caps the score at 3), then whether it follows the system prompt's voice, format and length.
10 is an answer the user would be delighted by; 5 is usable but flawed; 0 is wrong or off-topic.
Give a one-sentence reason naming the concrete strength or flaw."""

SCORE_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "integer", "minimum": 0, "maximum": 10}, "reason": {"type": "string"}},
    "required": ["score", "reason"],
}


@tool
def evaluate_model(ctx: Ctx, target: str = "current", reason: str = "") -> dict:
    """Score a model on the project's fixed test questions (set with update_project): each answer is
    graded 0–10 by GPT-6 against the goal and system prompt. The result is kept per checkpoint and
    shown on the canvas, so before/after and run-to-run comparisons are numbers. Run it on the base
    model before training (the score to beat), after every run, and before proposing an export.
    target: current (served) | base | checkpoint:<id>. Costs one judge call per question. Outside a
    round the user set in motion it's a proposal they confirm first; give a plain-words `reason`."""
    pid = ctx.context.project_id
    with Session(engine()) as s:
        st = studio_state(s, pid)  # first: creating the row commits, which would expire `project`
        previous = [dict(e) for e in st.evals]
        project = s.get(Project, pid)
        cases = db_test_cases(project)
    questions = [c["input"] for c in cases]
    if not cases:
        raise ValueError("No test cases yet: set them with update_project(test_cases=[...]) first.")
    if proposal := _spend(
        pid, "evaluate_model", "evaluate", f"Score the {target.split(':')[0]} model on {len(questions)} test questions",
        reason, {"questions": len(questions), "target": target}, {"target": target},
    ):  # fmt: skip
        return proposal
    judge = get_provider()
    items = []
    items = _score_all(project, judge, cases, target)
    mean = round(sum(i["score"] for i in items) / len(items), 1)
    checked = [i for i in items if i.get("exact") is not None]
    exact_rate = round(sum(1 for i in checked if i["exact"]) / len(checked), 2) if checked else None
    with Session(engine()) as s:
        project = s.get(Project, pid)
        if target == "base":
            checkpoint_id = None
        elif target.startswith("checkpoint:"):
            checkpoint_id = int(target.split(":", 1)[1])
        else:
            checkpoint_id = _served_checkpoint_id(s, project)
        record = {
            "id": f"e{int(time.time() * 1000)}",
            "target": "base" if checkpoint_id is None else f"checkpoint:{checkpoint_id}",
            "checkpoint_id": checkpoint_id,
            "mean": mean,
            "exact_rate": exact_rate,
            "items": items,
            "at": now().isoformat(),
        }
        st = studio_state(s, pid)
        st.evals = (previous + [record])[-60:]  # the whole history: disk is not the constraint
        flag_modified(st, "evals")
        st.stage = "evaluate"
        s.add(st)
        s.commit()
    canvas_changed(pid)
    best = max(previous, key=lambda e: e["mean"], default=None)
    return {
        "target": record["target"],
        "checkpoint_id": checkpoint_id,
        "mean_score": mean,
        "exact_match_rate": exact_rate,  # over cases with an expected output; None if there are none
        "scores": [
            {"q": _clip(i["prompt"], 80), "kind": i["kind"], "score": i["score"], "why": _clip(i["reason"], 120)}
            for i in items
        ],
        "previous_best": {k: best[k] for k in ("target", "checkpoint_id", "mean")} if best else None,
        "note": "Compare with the base score and the previous best; export the best-scoring checkpoint.",
    }


def _canon(text: str) -> tuple[str, bool]:
    """A comparable form of an output: canonical JSON when it parses, else trimmed lowercase text."""
    import json as _json

    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return _json.dumps(_json.loads(t), sort_keys=True, separators=(",", ":")), True
    except ValueError:
        return " ".join(t.lower().split()), False


def _score_all(project: Project, judge, cases: list[dict], target: str) -> list[dict]:
    """Exact match first (an expected output is the ground truth), the judge only where there is
    no expected output or the answer differs from it and may still deserve partial credit."""
    items = []
    for case in cases:
        _check_stop(project.id)
        q, expected = case["input"], case.get("expected")
        answer = _generate(project, q, 0.3, 300, target)["text"]
        item = {"prompt": q, "kind": case.get("kind", "on-goal"), "answer": answer, "exact": None}
        if expected:
            got, got_json = _canon(answer)
            want, want_json = _canon(expected)
            item["exact"] = got == want
            if item["exact"]:
                item |= {"score": 10, "reason": "matches the expected output"}
            elif want_json and not got_json:
                item |= {"score": 0, "reason": "not valid JSON, and JSON was expected"}
            else:
                v = judge.json(
                    SCORE_SYSTEM + "\nYou are also given the expected output: score by how close the answer is to it.",
                    f"Goal: {project.goal}\nSystem prompt: {project.system_prompt or '(none)'}\n\n"
                    f"Input: {q}\n\nExpected output:\n{expected}\n\nAnswer:\n{answer}",
                    SCORE_SCHEMA,
                )
                item |= {"score": max(0, min(9, int(v.get("score", 0)))), "reason": (v.get("reason") or "")[:300]}
        else:
            v = judge.json(
                SCORE_SYSTEM,
                f"Goal: {project.goal}\nSystem prompt: {project.system_prompt or '(none)'}\n\n"
                f"Question: {q}\n\nAnswer:\n{answer}",
                SCORE_SCHEMA,
            )
            item |= {"score": max(0, min(10, int(v.get("score", 0)))), "reason": (v.get("reason") or "")[:300]}
        items.append(item)
    return items


@tool
def serve_checkpoint(ctx: Ctx, checkpoint_id: int) -> dict:
    """Make the project serve an earlier checkpoint again (a roll-back). Use it when a round made
    the model worse (a val_worse or overfitting warning, or a lower evaluate_model score): later
    runs continue from the served checkpoint, and export packages it. get_status lists checkpoints."""
    pid = ctx.context.project_id
    from slm.train.jobs import model_in_use
    from slm.train.jobs import serve_checkpoint as _serve

    with Session(engine()) as s:
        if (busy := model_in_use(s, pid)) is not None:
            raise ValueError(f"Job {busy} is using this project's model; wait for it to finish.")
        project = s.get(Project, pid)
        c = s.get(Checkpoint, checkpoint_id)
        if c is None or c.project_id != pid:
            raise ValueError("no such checkpoint in this project")
        _serve(s, project, c)
        out = {"serving": f"checkpoint:{c.id}", "kind": c.kind, "job_id": c.job_id, "metrics": c.metrics}
    infer.unload()  # the next generation loads the rolled-back weights
    canvas_changed(pid)
    return out


@tool
def ask_user_to_compare(ctx: Ctx, prompts: list[str]) -> dict:
    """Put side-by-side A/B answers on the canvas for the *user* to judge. Only use this when the
    user has said they want to judge answers themselves; otherwise use ai_review_answers. You'll be
    told when they've finished."""
    pid = ctx.context.project_id
    with Session(engine()) as s:
        project = s.get(Project, pid)
    items = []
    for q in prompts[:10]:
        a, b = _two_answers(project, q, 300)
        items.append(
            {
                "id": f"c{int(time.time() * 1000)}{len(items)}",
                "prompt": q,
                "a": a,
                "b": b,
                "params_a": PARAMS_A,
                "params_b": PARAMS_B,
                "status": "pending",
            }
        )
    identical = sum(1 for i in items if i["a"] == i["b"])
    with Session(engine()) as s:
        st = studio_state(s, pid)
        st.comparisons = [dict(c) for c in st.comparisons if c.get("status") == "pending"] + items
        flag_modified(st, "comparisons")
        st.stage = "evaluate"
        s.add(st)
        s.commit()
    canvas_changed(pid)
    return {
        "queued": len(items),
        "identical_pairs": identical,
        "note": "Identical pairs carry no preference signal; the user can still rewrite the answer.",
    }


@tool
def feedback_summary(ctx: Ctx, limit: int = 15) -> dict:
    """The user's recent judgements and critiques, and how much training signal is ready."""
    pid = ctx.context.project_id
    with Session(engine()) as s:
        rows = s.exec(
            select(Feedback).where(Feedback.project_id == pid).order_by(Feedback.id.desc()).limit(limit)
        ).all()
    return {
        "recent": [
            {
                "prompt": _clip(f.prompt, 120),
                "choice": f.choice,
                "critique": f.critique,
                "rewrote": bool(f.edited_answer),
            }
            for f in rows
        ],
        "counts": {"judgements": len(rows)},
    }


@tool
def generate_synthetic_examples(ctx: Ctx, kind: str, count: int, focus: str, reason: str = "") -> dict:
    """Have the teacher model (GPT-6) write new training examples: kind "sft" (question + ideal
    answer) or "preference" (ideal vs. weak answer). Two uses:
    - no good public data: write a seed dataset from the goal and the user's example questions
      (100–200 sft examples with a focus covering the range of questions users will ask);
    - after feedback: target a weakness the user's critiques revealed.
    Examples wait unapproved; spot-check them with review_synthetic_examples, then approve.
    This costs API calls (one per example). Outside a round the user set in motion it's a proposal
    they confirm first; give a plain-words `reason` for the card."""
    pid = ctx.context.project_id
    count = max(1, min(count, 200))
    with Session(engine()) as s:
        unreviewed = sum(
            count_rows(s, m, m.project_id == pid, *awaiting_review(m)) for m in (PreferencePair, SftExample)
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
    return {
        "status": "prepared",
        "version_id": v.id,
        "train": v.n_train,
        "valid": v.n_valid,
        "cleaning": (v.cleaning_report or {}).get("dropped"),
        "tokens": {k: (v.token_stats or {}).get(k) for k in ("p50", "p95", "max", "estimated")},
        "length": (v.cleaning_report or {}).get("length"),
    }


JUDGE_SYSTEM = """You are an expert reviewer grading a small language model that is being fine-tuned
for a specific goal. For one prompt you see two candidate answers, A and B, from the model.
Decide which is better for the goal (accuracy first, then following the system prompt's style),
or "tie" if equally good, or "both_bad" if neither is acceptable. Then write the ideal answer, in
the exact style the goal and system prompt ask for, and a one-sentence critique naming the concrete
flaws. If the better answer is already ideal, repeat it verbatim as the ideal answer."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "choice": {"type": "string", "enum": ["a", "b", "tie", "both_bad"]},
        "ideal_answer": {"type": "string"},
        "critique": {"type": "string"},
    },
}


@tool
def ai_review_answers(ctx: Ctx, prompts: list[str], reason: str = "") -> dict:
    """Stand in for the human judge. For each prompt the local model writes two answers (A and B);
    GPT-6 picks the better one, writes the ideal answer and critiques the flaws. Each verdict
    becomes training signal automatically: a preference pair (better vs. worse, for DPO) and, when
    the answers were flawed, a corrected example (for SFT). Use 8–12 varied, realistic prompts
    that did NOT come from the training data. Results show on the canvas.
    This costs GPU time and API calls. Outside a round the user set in motion it's a proposal they
    confirm first; give a plain-words `reason` for the card."""
    pid = ctx.context.project_id
    if proposal := _spend(
        pid, "ai_review_answers", "review", f"Have GPT-6 review {len(prompts[:12])} answers", reason,
        {"prompts": len(prompts[:12])}, {"prompts": prompts[:12]},
    ):  # fmt: skip
        return proposal
    with Session(engine()) as s:
        project = s.get(Project, pid)
    judge = get_provider()
    verdicts = []
    for q in prompts[:12]:
        _check_stop(pid)
        a, b = _two_answers(project, q, 350)
        v = judge.json(
            JUDGE_SYSTEM,
            f"Goal: {project.goal}\nSystem prompt: {project.system_prompt or '(none)'}\n\nPrompt: {q}\n\n"
            f"Answer A:\n{a}\n\nAnswer B:\n{b}",
            JUDGE_SCHEMA,
        )
        choice = v.get("choice", "tie")
        chosen = a if choice == "a" else b if choice == "b" else ""
        ideal = (v.get("ideal_answer") or "").strip()
        # Only store the ideal answer as a correction when it differs from what the model said.
        edited = ideal if ideal and ideal != chosen.strip() else ""
        if choice == "both_bad" and not edited:
            edited = ideal or ""
        out = record_feedback(
            pid,
            prompt=q,
            system=project.system_prompt,
            candidate_a=a,
            candidate_b=b,
            choice=choice,
            edited_answer=edited,
            critique=v.get("critique", "") or "reviewed by AI",
            params_a=PARAMS_A,
            params_b=PARAMS_B,
            model_ref="ai-judge",
        )
        verdicts.append(
            {
                "id": f"ai{out['feedback_id']}",
                "prompt": q,
                "a": a,
                "b": b,
                "status": "judged",
                "judge": "ai",
                "choice": choice,
                "critique": v.get("critique", ""),
                "ideal": edited,
                "pairs": out["preference_pairs"],
                "sft": out["sft_examples"],
            }
        )
    with Session(engine()) as s:
        st = studio_state(s, pid)
        st.comparisons = [dict(c) for c in st.comparisons][-20:] + verdicts
        flag_modified(st, "comparisons")
        st.stage = "refine"
        s.add(st)
        s.commit()
    canvas_changed(pid)
    return {
        "reviewed": len(verdicts),
        "verdicts": dict(Counter(v["choice"] for v in verdicts)),
        "preference_pairs_added": sum(v["pairs"] for v in verdicts),
        "corrected_examples_added": sum(v["sft"] for v in verdicts),
        "critiques": [_clip(v["critique"], 160) for v in verdicts],
    }


@tool
def finish_project(ctx: Ctx, summary: str) -> str:
    """Declare the work done, once the model is exported (or you've decided it can't get better).
    Stops autopilot until the user wants more: they can always keep improving the model later.
    summary: two or three sentences on what was built and how good it is."""
    pid = ctx.context.project_id
    with Session(engine()) as s:
        st = studio_state(s, pid)
        st.completed, st.stage, st.note = True, "export", summary[:200]
        s.add(st)
        s.commit()
    canvas_changed(pid)
    return "finished"


# ── the whole machine ───────────────────────────────────────────────────────


@tool
def machine_overview(ctx: Ctx) -> dict:
    """What this Mac is doing across ALL projects: the training run using the GPU (project,
    progress, minutes left), the queue of training runs waiting, and every project's state.
    Only one model trains at a time on this Mac; others queue. Check this before starting a
    training run, and tell the user if theirs will wait behind another project (and roughly how long)."""
    data = overview()
    data["this_project_id"] = ctx.context.project_id
    return data


@tool
def manage_project(ctx: Ctx, project_id: int, action: str) -> dict:
    """Pause, resume or stop ANOTHER project (or this one). action: pause (stop its autopilot; a
    running job finishes), stop (pause and cancel its queued/running jobs, freeing the GPU), resume
    (turn its autopilot back on). Only act on another project when the user asks you to, e.g. to
    free the GPU for this one; explain what will happen first (a stopped training run is lost)."""
    if action == "resume":
        return resume_project(project_id)
    if action in ("pause", "stop"):
        return stop_project(project_id, cancel_jobs=action == "stop")
    raise ValueError("action must be pause, stop or resume")


# ── the person ──────────────────────────────────────────────────────────────


@tool
def set_user_level(ctx: Ctx, level: str, evidence: str) -> dict:
    """Record how much ML the user knows: beginner | intermediate | expert. Infer it from how they
    write: vocabulary, what they ask about, what they want to control. For example, asking about
    LoRA rank, learning-rate schedules or DPO β suggests expert; "what's a model?" suggests
    beginner. evidence: one short line saying why. It's remembered for all their future projects,
    and you'll adapt your explanations to it. Update it if new evidence contradicts it."""
    p = profile.set_level(level, evidence)
    return {"level": p.level, "evidence": p.level_evidence}


@tool
def remember_about_user(ctx: Ctx, text: str, kind: str = "preference") -> dict:
    """Remember something durable about the user for all their future projects. kind: preference
    (e.g. "prefers the smallest model that works", "wants 4-bit exports", "likes to judge answers
    themselves"), fact (e.g. "teaches high-school physics"), or goal (e.g. "wants to ship models
    to an iPhone app"). Only things that will still be true next time; not project details."""
    return profile.remember(text, kind, ctx.context.project_id)


# ── export ──────────────────────────────────────────────────────────────────


@tool
def export_model(ctx: Ctx, name: str, quantize_bits: int | None = None, reason: str = "") -> dict:
    """Propose packaging the current model: fuse adapters, optionally quantize (4/6/8; skip if the
    base is already quantized), and write a model card. It runs only once the user confirms it on
    a card. reason: one plain sentence for the card. Afterwards they can chat with the exported
    model on the "Try it" page."""
    from slm.export.fuse import check_bits

    check_bits(quantize_bits)
    config = {"name": name, "quantize_bits": quantize_bits, "sampling": {"temperature": 0.7, "top_p": 0.95}}
    details = {"name": name, "quantize_bits": quantize_bits}
    return confirm.propose(ctx.context.project_id, "export", f"Export the model as “{name}”", reason, details,
                           {"config": config})  # fmt: skip


ALL_TOOLS = [
    get_status,
    set_stage,
    update_project,
    find_base_models,
    choose_base_model,
    search_datasets,
    preview_dataset,
    scout_datasets,
    plan_preparation,
    import_dataset,
    inspect_dataset,
    prepare_dataset,
    plan_training,
    start_training,
    training_progress,
    cancel_job,
    try_model,
    evaluate_model,
    serve_checkpoint,
    ask_user_to_compare,
    feedback_summary,
    generate_synthetic_examples,
    review_synthetic_examples,
    build_dataset_from_examples,
    export_model,
    ai_review_answers,
    finish_project,
    machine_overview,
    manage_project,
    set_user_level,
    remember_about_user,
]
