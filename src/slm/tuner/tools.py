"""The Tuner's hands: every action it can take on the platform.

Each tool wraps existing, tested backend code and returns a compact dict the agent can reason
about. Quick jobs (dataset import, preparation, synthesis) are awaited inside the tool. Long ones
(downloads, training, export) return a job id at once; the Tuner is woken when they finish.
"""

import asyncio
import functools
import time
from dataclasses import dataclass
from pathlib import Path

from agents import RunContextWrapper, function_tool
from sqlalchemy.orm.attributes import flag_modified
from sqlmodel import Session, func, select

from slm import hardware
from slm.data import format as fmt
from slm.data import scout_tools
from slm.data.pipeline import preview_mapping
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
    StudioState,
    engine,
    now,
)
from slm.events import bus
from slm.models import hub, manage
from slm.train.config import TrainConfig, preset
from slm.train.worker import worker

STAGES = ("goal", "model", "data", "train", "evaluate", "refine", "export")
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


def canvas_changed(project_id: int) -> None:
    bus.publish(f"tuner:{project_id}", {"type": "canvas"})


def _studio(s: Session, project_id: int) -> StudioState:
    st = s.get(StudioState, project_id)
    if st is None:
        st = StudioState(project_id=project_id)
        s.add(st)
        s.commit()
        s.refresh(st)
    return st


def _submit(kind: str, config: dict, project_id: int) -> Job:
    job = worker.submit(kind, config | {"origin": "tuner", "notify": kind in NOTIFY_KINDS}, project_id)
    canvas_changed(project_id)
    return job


def _wait(job_id: int, timeout: float) -> Job:
    """Wait for a quick job; the UI shows its progress meanwhile."""
    deadline = time.time() + timeout
    while True:
        with Session(engine()) as s:
            job = s.get(Job, job_id)
            if job.status in ("succeeded", "failed", "cancelled") or time.time() > deadline:
                return job
        time.sleep(1)


def _model_path(project: Project) -> str | None:
    return project.current_model_path or (manage.local_path_for(project.base_model) if project.base_model else None)


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


@tool
def get_status(ctx: Ctx) -> dict:
    """Everything about the project right now: hardware, goal, base model, datasets, prepared
    versions, running and recent jobs, trained checkpoints (with metrics and warnings), feedback
    counts and exports. Call this first in every new turn if you're unsure where things stand."""
    pid = ctx.context.project_id
    hw = hardware.detect()
    with Session(engine()) as s:
        p = s.get(Project, pid)
        st = _studio(s, pid)
        datasets = s.exec(select(Dataset).where(Dataset.project_id == pid)).all()
        versions = s.exec(select(DatasetVersion).where(DatasetVersion.project_id == pid)).all()
        jobs = s.exec(select(Job).where(Job.project_id == pid).order_by(Job.id.desc()).limit(8)).all()
        ckpts = s.exec(select(Checkpoint).where(Checkpoint.project_id == pid).order_by(Checkpoint.id)).all()
        job_results = {j.id: j.result for j in s.exec(select(Job).where(Job.id.in_([c.job_id for c in ckpts]))).all()}

        def count(model, *where) -> int:
            return s.exec(select(func.count()).select_from(model).where(*where)).one()

        feedback = {
            "judgements": count(Feedback, Feedback.project_id == pid),
            "preference_pairs_ready": count(
                PreferencePair,
                PreferencePair.project_id == pid,
                PreferencePair.approved == True,  # noqa: E712
                PreferencePair.used_in_job_id == None,  # noqa: E711
            ),
            "sft_examples_ready": count(
                SftExample,
                SftExample.project_id == pid,
                SftExample.approved == True,  # noqa: E712
                SftExample.used_in_job_id == None,  # noqa: E711
            ),
            "synthetic_awaiting_review": count(
                PreferencePair, PreferencePair.project_id == pid, PreferencePair.approved == False
            )  # noqa: E712
            + count(SftExample, SftExample.project_id == pid, SftExample.approved == False),  # noqa: E712
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
            "project": {"name": p.name, "goal": p.goal, "system_prompt": p.system_prompt},
            "models_already_on_this_mac": [
                {k: m[k] for k in ("repo_id", "params_b", "bits", "fit")} for m in manage.local_models()
            ],
            "canvas_stage": st.stage,
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
                    "serving": p.current_adapter_path == c.adapter_path,
                }
                for i, c in enumerate(ckpts)
            ],
            "feedback": feedback,
            "exports": [
                {
                    "path": j.result.get("path"),
                    "size_gb": j.result.get("size_gb"),
                    "min_ram_gb": j.result.get("min_ram_gb"),
                }
                for j in exports
            ],
        }


@tool
def set_stage(ctx: Ctx, stage: str, note: str) -> str:
    """Move the live canvas to a pipeline stage and show one short line about what's happening.
    stage: goal | model | data | train | evaluate | refine | export. Call it whenever the work moves on."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    with Session(engine()) as s:
        st = _studio(s, ctx.context.project_id)
        st.stage, st.note, st.updated_at = stage, note[:200], now()
        s.add(st)
        s.commit()
    canvas_changed(ctx.context.project_id)
    return f"canvas → {stage}"


@tool
def update_project(
    ctx: Ctx, name: str | None = None, goal: str | None = None, system_prompt: str | None = None
) -> dict:
    """Record the project's name, the model's goal (one or two sentences), and the system prompt
    the model will be trained and used with. Set these as soon as you understand what the user wants."""
    with Session(engine()) as s:
        p = s.get(Project, ctx.context.project_id)
        if name:
            p.name = name[:60]
        if goal:
            p.goal = goal
        if system_prompt is not None:
            p.system_prompt = system_prompt
        s.add(p)
        s.commit()
        s.refresh(p)
    canvas_changed(ctx.context.project_id)
    return {"name": p.name, "goal": p.goal, "system_prompt": p.system_prompt}


# ── base model ──────────────────────────────────────────────────────────────


@tool
def find_base_models(ctx: Ctx, query: str, max_params_billion: float = 3.0) -> list[dict]:
    """Search for MLX base models that fit this Mac, smallest first. Models already downloaded on
    this Mac are marked and listed first (they cost nothing to use). Returns size, bits, estimated
    training memory and a fit verdict. Prefer the smallest Instruct model that can do the job."""
    local = {m["repo_id"]: m for m in manage.local_models()}
    rows = hub.search_models(query, max_params_b=max_params_billion, limit=12)
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
    return sorted(found.values(), key=lambda m: (not m["already_on_this_mac"], m["params_b"]))


@tool
def choose_base_model(ctx: Ctx, repo_id: str) -> dict:
    """Set the project's base model and download it (runs in the background; you'll be told when
    it's done). Only possible before any training has happened."""
    pid = ctx.context.project_id
    with Session(engine()) as s:
        p = s.get(Project, pid)
        if p.base_model != repo_id and s.exec(select(Checkpoint).where(Checkpoint.project_id == pid)).first():
            raise ValueError("Training has already started on another base model; start a new project to switch.")
        p.base_model = repo_id
        p.current_adapter_path = None
        local = manage.local_path_for(repo_id)
        if local is None and (rec := manage.register_local(repo_id)) is not None:
            local = rec.local_path  # already in the HF cache: no download needed
        p.current_model_path = local
        s.add(p)
        s.commit()
    canvas_changed(pid)
    if local:
        return {"status": "already downloaded", "repo_id": repo_id}
    job = _submit("download", {"repo_id": repo_id}, pid)
    return {"status": "downloading", "job_id": job.id, "note": "You'll receive a job update when it finishes."}


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
def import_dataset(
    ctx: Ctx, repo_id: str, max_rows: int = 3000, config: str | None = None, split: str = "train"
) -> dict:
    """Import rows from a Hugging Face dataset into the project. A few thousand good rows is plenty
    for a small model. Waits for the import to finish."""
    job = _submit(
        "import_dataset",
        {"repo_id": repo_id, "max_rows": max_rows, "config": config, "split": split},
        ctx.context.project_id,
    )
    job = _wait(job.id, 300)
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
    min_answer_chars: int = 2,
    long_examples: str = "auto",
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
            "rules": {"min_chars": min_answer_chars, "long_examples": long_examples},
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
            "sample_record": _clip(preview_mapping(ds, mapping, n=1)["records"][:1], 500),
        }


# ── training ────────────────────────────────────────────────────────────────


def _config(
    project: Project, mode: str, preset_name: str, version: DatasetVersion | None, overrides: dict
) -> tuple[TrainConfig, dict]:
    path = _model_path(project)
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
        info |= {
            "iterations": cfg.total_iters(version.n_train),
            "epochs": cfg.epochs_for(version.n_train),
            "train_examples": version.n_train,
        }
    return cfg, info


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
    pid = ctx.context.project_id
    with Session(engine()) as s:
        project = s.get(Project, pid)
        version = s.get(DatasetVersion, dataset_version_id) if dataset_version_id else None
    overrides = dict(
        learning_rate=learning_rate,
        epochs=epochs,
        iters=iters,
        lora_rank=lora_rank,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
    )
    _, info = _config(project, mode, preset_name, version, overrides)
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
) -> dict:
    """Start a training run in the background (you'll be told when it finishes; progress shows live
    on the canvas). mode sft needs dataset_version_id; mode dpo trains on the user's feedback pairs
    (needs at least 3). start_from: current (continue from the served model) | base (fresh run).
    Confirm with the user before starting, since runs take minutes."""
    pid = ctx.context.project_id
    if mode not in ("sft", "dpo"):
        raise ValueError("mode must be sft or dpo")
    with Session(engine()) as s:
        project = s.get(Project, pid)
        version = s.get(DatasetVersion, dataset_version_id) if dataset_version_id else None
    if mode == "sft" and version is None:
        raise ValueError("SFT needs dataset_version_id (prepare a dataset first)")
    overrides = dict(
        learning_rate=learning_rate,
        epochs=epochs,
        iters=iters,
        lora_rank=lora_rank,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
    )
    cfg, info = _config(project, mode, preset_name, version, overrides)
    if not info["fits"]:
        return {"status": "won't fit", **info, "hint": "use preset safe, lower batch_size or max_seq_length"}
    config = {"train": cfg.model_dump(), "start_from": start_from}
    if version:
        config["dataset_version_id"] = version.id
    job = _submit(mode, config, pid)
    from slm.sessions import overview

    gpu = overview()["capacity"]
    ahead = gpu["gpu_running"]
    if ahead and ahead["job_id"] != job.id:
        info["queued_behind"] = ahead | {"waiting_before_this": len(gpu["gpu_queue"]) - 1}
    with Session(engine()) as s:
        st = _studio(s, pid)
        st.stage = "train" if mode == "sft" else "refine"
        s.add(st)
        s.commit()
    canvas_changed(pid)
    return {"status": "started", "job_id": job.id, **info}


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


def _generate(
    project: Project, prompt: str, temperature: float, max_tokens: int, target: str = "current", seed: int | None = None
) -> dict:
    from slm.inference.engine import EngineBusy, SamplingParams
    from slm.inference.engine import engine as infer

    path = _model_path(project)
    if not path:
        raise ValueError("No model downloaded yet")
    where = {"model_path": path, "adapter_path": project.current_adapter_path}
    if target == "base":
        where = {"model_path": manage.local_path_for(project.base_model), "adapter_path": None}
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


@tool
def try_model(
    ctx: Ctx, prompts: list[str], target: str = "current", temperature: float = 0.7, max_tokens: int = 300
) -> list[dict]:
    """Ask the local model some questions and see its answers (also shown on the canvas).
    target: current (latest trained) | base (the untrained model, for before/after comparisons).
    Use 3–5 varied prompts that reflect the goal, including ones not in the training data."""
    pid = ctx.context.project_id
    with Session(engine()) as s:
        project = s.get(Project, pid)
    results = []
    for q in prompts[:6]:
        r = _generate(project, q, temperature, max_tokens, target)
        results.append({"prompt": q, "target": target, **r})
    with Session(engine()) as s:
        st = _studio(s, pid)
        st.samples = (results + list(st.samples))[:12]
        flag_modified(st, "samples")
        s.add(st)
        s.commit()
    canvas_changed(pid)
    return results


@tool
def ask_user_to_compare(ctx: Ctx, prompts: list[str]) -> dict:
    """Put side-by-side A/B answers on the canvas for the *user* to judge. Only use this when the
    user has said they want to judge answers themselves; otherwise use ai_review_answers. You'll be
    told when they've finished."""
    import random

    pid = ctx.context.project_id
    with Session(engine()) as s:
        project = s.get(Project, pid)
    items = []
    for q in prompts[:10]:
        a = _generate(project, q, 0.7, 300, seed=random.randint(0, 10**9))
        b = _generate(project, q, 1.05, 300, seed=random.randint(0, 10**9))
        items.append(
            {
                "id": f"c{int(time.time() * 1000)}{len(items)}",
                "prompt": q,
                "a": a["text"],
                "b": b["text"],
                "params_a": {"temperature": 0.7},
                "params_b": {"temperature": 1.05},
                "status": "pending",
            }
        )
    identical = sum(1 for i in items if i["a"] == i["b"])
    with Session(engine()) as s:
        st = _studio(s, pid)
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
def generate_synthetic_examples(ctx: Ctx, kind: str, count: int, focus: str) -> dict:
    """Have the teacher model (GPT-6) write new training examples: kind "sft" (question + ideal
    answer) or "preference" (ideal vs. weak answer). Two uses:
    - no good public data: write a seed dataset from the goal and the user's example questions
      (100–200 sft examples with a focus covering the range of questions users will ask);
    - after feedback: target a weakness the user's critiques revealed.
    Examples wait unapproved; spot-check them with review_synthetic_examples, then approve."""
    job = _submit(
        "synthesize", {"kind": kind, "count": max(1, min(count, 200)), "focus": focus}, ctx.context.project_id
    )
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
                for row in s.exec(select(model).where(model.project_id == pid, model.approved == False)).all():  # noqa: E712
                    if (model, row.id) not in rejected:
                        row.approved = True
                        s.add(row)
                        changed += 1
        s.commit()
        pairs = s.exec(
            select(PreferencePair).where(PreferencePair.project_id == pid, PreferencePair.approved == False).limit(show)
        ).all()  # noqa: E712
        sft = s.exec(
            select(SftExample).where(SftExample.project_id == pid, SftExample.approved == False).limit(show)
        ).all()  # noqa: E712
        pending = len(
            s.exec(select(SftExample.id).where(SftExample.project_id == pid, SftExample.approved == False)).all()
        ) + len(  # noqa: E712
            s.exec(
                select(PreferencePair.id).where(PreferencePair.project_id == pid, PreferencePair.approved == False)
            ).all()  # noqa: E712
        )
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
    synthetic examples) into a prepared dataset version you can train on."""
    from slm.data.pipeline import build_feedback_version

    pid = ctx.context.project_id
    with Session(engine()) as s:
        project = s.get(Project, pid)
    v = build_feedback_version(pid, model_path=_model_path(project), max_seq_length=max_seq_length)
    canvas_changed(pid)
    return {
        "status": "prepared",
        "version_id": v.id,
        "train": v.n_train,
        "valid": v.n_valid,
        "cleaning": (v.cleaning_report or {}).get("dropped"),
        "tokens": {k: (v.token_stats or {}).get(k) for k in ("p50", "p95", "max")},
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
def ai_review_answers(ctx: Ctx, prompts: list[str]) -> dict:
    """Stand in for the human judge. For each prompt the local model writes two answers (A and B);
    GPT-6 picks the better one, writes the ideal answer and critiques the flaws. Each verdict
    becomes training signal automatically: a preference pair (better vs. worse, for DPO) and, when
    the answers were flawed, a corrected example (for SFT). Use 8–12 varied, realistic prompts
    that did NOT come from the training data. Results show on the canvas."""
    import random

    from slm.agents.provider import get_provider
    from slm.feedback import record_feedback

    pid = ctx.context.project_id
    with Session(engine()) as s:
        project = s.get(Project, pid)
    judge = get_provider()
    verdicts = []
    for q in prompts[:12]:
        a = _generate(project, q, 0.7, 350, seed=random.randint(0, 10**9))["text"]
        b = _generate(project, q, 1.05, 350, seed=random.randint(0, 10**9))["text"]
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
            params_a={"temperature": 0.7},
            params_b={"temperature": 1.05},
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
        st = _studio(s, pid)
        st.comparisons = [dict(c) for c in st.comparisons][-20:] + verdicts
        flag_modified(st, "comparisons")
        st.stage = "refine"
        s.add(st)
        s.commit()
    canvas_changed(pid)
    from collections import Counter

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
    Stops autopilot. summary: two or three sentences on what was built and how good it is."""
    pid = ctx.context.project_id
    with Session(engine()) as s:
        st = _studio(s, pid)
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
    from slm.sessions import overview

    data = overview()
    data["this_project_id"] = ctx.context.project_id
    return data


@tool
def manage_project(ctx: Ctx, project_id: int, action: str) -> dict:
    """Pause, resume or stop ANOTHER project (or this one). action: pause (stop its autopilot; a
    running job finishes), stop (pause and cancel its queued/running jobs, freeing the GPU), resume
    (turn its autopilot back on). Only act on another project when the user asks you to, e.g. to
    free the GPU for this one; explain what will happen first (a stopped training run is lost)."""
    from slm.sessions import resume_project, stop_project

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
    from slm import profile

    p = profile.set_level(level, evidence)
    return {"level": p.level, "evidence": p.level_evidence}


@tool
def remember_about_user(ctx: Ctx, text: str, kind: str = "preference") -> dict:
    """Remember something durable about the user for all their future projects. kind: preference
    (e.g. "prefers the smallest model that works", "wants 4-bit exports", "likes to judge answers
    themselves"), fact (e.g. "teaches high-school physics"), or goal (e.g. "wants to ship models
    to an iPhone app"). Only things that will still be true next time; not project details."""
    from slm import profile

    return profile.remember(text, kind, ctx.context.project_id)


# ── export ──────────────────────────────────────────────────────────────────


@tool
def export_model(ctx: Ctx, name: str, quantize_bits: int | None = None) -> dict:
    """Package the current model: fuse adapters, optionally quantize (4/6/8; skip if the base is
    already quantized), and write a model card. Runs in the background."""
    job = _submit(
        "export",
        {"name": name, "quantize_bits": quantize_bits, "sampling": {"temperature": 0.7, "top_p": 0.95}},
        ctx.context.project_id,
    )
    return {"status": "exporting", "job_id": job.id}


ALL_TOOLS = [
    get_status,
    set_stage,
    update_project,
    find_base_models,
    choose_base_model,
    search_datasets,
    preview_dataset,
    import_dataset,
    inspect_dataset,
    prepare_dataset,
    plan_training,
    start_training,
    training_progress,
    cancel_job,
    try_model,
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
